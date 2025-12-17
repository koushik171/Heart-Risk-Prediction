import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { MapPin, Phone, Clock, Star, Navigation } from 'lucide-react';

interface Hospital {
  id: string;
  name: string;
  address: string;
  phone: string;
  distance: number;
  rating: number;
  specialties: string[];
  emergency: boolean;
  hours: string;
}

const mockHospitals: Hospital[] = [
  {
    id: '1',
    name: 'City General Hospital',
    address: '123 Main St, Downtown',
    phone: '+1 (555) 123-4567',
    distance: 2.3,
    rating: 4.5,
    specialties: ['Cardiology', 'Emergency', 'ICU'],
    emergency: true,
    hours: '24/7'
  },
  {
    id: '2',
    name: 'Heart Care Medical Center',
    address: '456 Oak Ave, Medical District',
    phone: '+1 (555) 234-5678',
    distance: 3.7,
    rating: 4.8,
    specialties: ['Cardiology', 'Cardiac Surgery', 'Interventional Cardiology'],
    emergency: false,
    hours: '6:00 AM - 10:00 PM'
  },
  {
    id: '3',
    name: 'Regional Medical Hospital',
    address: '789 Pine Rd, Northside',
    phone: '+1 (555) 345-6789',
    distance: 5.1,
    rating: 4.2,
    specialties: ['Cardiology', 'Emergency', 'Cardiac Rehabilitation'],
    emergency: true,
    hours: '24/7'
  }
];

const HospitalFinder = () => {
  const [location, setLocation] = useState('');
  const [hospitals, setHospitals] = useState<Hospital[]>([]);
  const [loading, setLoading] = useState(false);
  const [userCoords, setUserCoords] = useState<{lat: number, lng: number} | null>(null);

  const getCurrentLocation = () => {
    setLoading(true);
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          setUserCoords({ lat: latitude, lng: longitude });
          setLocation(`${latitude.toFixed(4)}, ${longitude.toFixed(4)}`);
          // Filter hospitals within 100km range
          const nearbyHospitals = mockHospitals.filter(hospital => hospital.distance <= 100);
          setHospitals(nearbyHospitals);
          setLoading(false);
        },
        () => {
          setLocation('Current Location');
          // Filter hospitals within 100km range
          const nearbyHospitals = mockHospitals.filter(hospital => hospital.distance <= 100);
          setHospitals(nearbyHospitals);
          setLoading(false);
        }
      );
    } else {
      setLocation('Location not supported');
      setHospitals(mockHospitals);
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    setLoading(true);
    setTimeout(() => {
      setHospitals(mockHospitals);
      setLoading(false);
    }, 1000);
  };

  const handleGetDirections = (hospital: Hospital) => {
    // Get real directions from user location to hospital
    if (userCoords) {
      // Use Google Maps directions with proper routing
      const directionsUrl = `https://www.google.com/maps/dir/?api=1&origin=${userCoords.lat},${userCoords.lng}&destination=${encodeURIComponent(hospital.address)}&travelmode=driving`;
      window.open(directionsUrl, '_blank');
    } else {
      // Try to get current location first, then redirect
      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            const { latitude, longitude } = position.coords;
            const directionsUrl = `https://www.google.com/maps/dir/?api=1&origin=${latitude},${longitude}&destination=${encodeURIComponent(hospital.address)}&travelmode=driving`;
            window.open(directionsUrl, '_blank');
          },
          () => {
            // Fallback to hospital location only
            const query = encodeURIComponent(hospital.address);
            window.open(`https://maps.google.com/?q=${query}`, '_blank');
          }
        );
      } else {
        // Fallback to hospital location only
        const query = encodeURIComponent(hospital.address);
        window.open(`https://maps.google.com/?q=${query}`, '_blank');
      }
    }
  };

  const handleCall = (phone: string) => {
    window.open(`tel:${phone}`);
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <MapPin className="h-6 w-6 text-primary" />
            <span>Find Nearby Hospitals</span>
          </CardTitle>
          <CardDescription>
            Locate cardiologists and hospitals with heart care services in your area
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="flex space-x-2">
              <Input
                placeholder="Enter your location (city, zip code, or address)"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                className="flex-1"
              />
              <Button onClick={handleSearch} disabled={loading || !location.trim()}>
                {loading ? 'Searching...' : 'Search'}
              </Button>
            </div>
            <Button 
              onClick={getCurrentLocation} 
              disabled={loading}
              variant="outline"
              className="w-full"
            >
              <MapPin className="h-4 w-4 mr-2" />
              {loading ? 'Getting Location...' : userCoords ? 'Location Found ✓' : 'Use My Current Location'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {hospitals.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">
            Found {hospitals.length} hospitals near you
          </h3>
          
          {hospitals.map((hospital) => (
            <Card key={hospital.id} className="hover:shadow-md transition-shadow">
              <CardContent className="p-6">
                <div className="flex justify-between items-start mb-4">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <h4 className="text-lg font-semibold">{hospital.name}</h4>
                      {hospital.emergency && (
                        <Badge variant="destructive" className="text-xs">
                          Emergency
                        </Badge>
                      )}
                    </div>
                    
                    <div className="space-y-2 text-sm text-muted-foreground">
                      <div className="flex items-center space-x-2">
                        <MapPin className="h-4 w-4" />
                        <span>{hospital.address}</span>
                        <Badge variant="outline" className="ml-2">
                          {hospital.distance} km away
                        </Badge>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <Phone className="h-4 w-4" />
                        <span>{hospital.phone}</span>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <Clock className="h-4 w-4" />
                        <span>{hospital.hours}</span>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                        <span>{hospital.rating}/5.0 rating</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mb-4">
                  <p className="text-sm font-medium mb-2">Specialties:</p>
                  <div className="flex flex-wrap gap-2">
                    {hospital.specialties.map((specialty, index) => (
                      <Badge key={index} variant="secondary" className="text-xs">
                        {specialty}
                      </Badge>
                    ))}
                  </div>
                </div>

                <div className="flex space-x-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleGetDirections(hospital)}
                    className="flex-1"
                  >
                    <Navigation className="h-4 w-4 mr-2" />
                    Directions
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleCall(hospital.phone)}
                    className="flex-1"
                  >
                    <Phone className="h-4 w-4 mr-2" />
                    Call
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {hospitals.length === 0 && !loading && (
        <Card>
          <CardContent className="flex items-center justify-center h-64">
            <div className="text-center space-y-3">
              <MapPin className="h-12 w-12 text-muted-foreground mx-auto" />
              <div>
                <p className="text-muted-foreground">
                  Enter your location to find nearby hospitals
                </p>
                <p className="text-sm text-muted-foreground">
                  We'll show you hospitals with cardiology services in your area
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default HospitalFinder;